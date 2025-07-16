//older version from grok history 
"use client";
import { Download, Eye, FileText, Presentation } from "lucide-react"
import html2canvas from "html2canvas"
import React, { useState } from 'react';
import { produce } from 'immer';
import { SearchResults } from './search-results';

// Define types for the query structure
type QueryNode = Keyword | Group;

interface Keyword {
  type: 'keyword';
  word: string;
  rule_op: 'all' | 'any';
  field: string;
}

interface Group {
  type: 'group';
  operator: 'AND' | 'OR';
  children: QueryNode[];
}

// Inline styles
const groupStyle: React.CSSProperties = {
  marginLeft: 0,
  marginBottom: '18px',
  padding: '18px 18px 14px 12px',
  border: '2px solid #BDD248',
  borderRadius: '8px',
  backgroundColor: '#f9f9f9',
  position: 'relative',
  display: 'flex',
  flexDirection: 'row',
  width: '100%',
  minWidth: 0,
  boxSizing: 'border-box',
  alignSelf: 'stretch',
};

const groupLineContainerStyle: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  marginRight: '32px',
  position: 'relative',
  minWidth: '54px',
  borderRadius: '10px',
  
};

const groupLineStyle: React.CSSProperties = {
  width: '2px',
  backgroundColor: '#111',
  flexGrow: 1,
  marginTop: '8px',
  marginBottom: '8px',
};

const groupOperatorLabelStyle: React.CSSProperties = {
  backgroundColor: '#fff',
  color: '#111',
  fontWeight: 'bold',
  padding: '2px 8px',
  borderRadius: '10px',
  border: '2px solid #111',
  position: 'absolute',
  left: '50%',
  transform: 'translate(-50%, -50%)',
  top: '50%',
  zIndex: 2,
  fontSize: '0.95em',
};

const rowStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: '6px',
  marginBottom: '10px',
  width: '100%',
  minWidth: 0,
  flexWrap: 'wrap',
};

const baseInputStyle: React.CSSProperties = {
  backgroundColor: 'white',
  border: 'none',
  borderBottom: '2px solid #bdbdbd',
  padding: '10px 10px',
  width: '100%',
  minWidth: 0,
  borderRadius: '10px',
  marginRight: 0,
  flex: 1,
  fontSize: '1.13em',
  outline: 'none',
  transition: 'border-bottom-color 0.18s cubic-bezier(.4,0,.2,1), border-bottom-width 0.18s cubic-bezier(.4,0,.2,1), background 0.18s cubic-bezier(.4,0,.2,1)',
};

const selectStyle: React.CSSProperties = {
  backgroundColor: 'white',
  border: '1px solid gray',
  padding: '10px 10px',
  marginRight: '4px',
  borderRadius: '10px',
  flex: 1,
  fontSize: '1.13em', 
  color: 'black',
  transition: 'background 0.18s cubic-bezier(.4,0,.2,1)',
};

const buttonStyle: React.CSSProperties = {
  backgroundColor: '#111',
  color: 'white',
  padding: '8px 18px',
  border: 'none',
  borderRadius: '15px',
  cursor: 'pointer',
  marginRight: 0,
  fontWeight: 600,
  fontSize: '1em',
  transition: 'background 0.2s',
};

const deleteButtonStyle: React.CSSProperties = {
  backgroundColor: '#f9f9f9',
  color: '#111',
  padding: 0,
  border: '2px solid #111',
  borderRadius: '50%',
  width: '28px',
  height: '28px',
  cursor: 'pointer',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  marginLeft: '6px',
  fontWeight: 700,
};

const operatorStyle: React.CSSProperties = {
  marginRight: '10px',
  fontWeight: 'bold',
};

// Main component
const AdvancedSearchBar: React.FC = () => {
  const [backendPort, setBackendPort] = React.useState<string | null>(null);
  React.useEffect(() => {
    fetch('/backend_port.txt')
      .then(res => res.text())
      .then(port => setBackendPort(port.trim()))
      .catch(() => setBackendPort(null));
  }, []);
  const [rootGroup, setRootGroup] = useState<Group>({
    type: 'group',
    operator: 'AND',
    children: [{ type: 'keyword', word: '', rule_op: 'all', field: 'title' }],
  });
  const [results, setResults] = useState<any[] | null>(null);
  const [hasSearched, setHasSearched] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Handle actions (add child, convert to group, update keyword/operator, delete node)
  const handleAction = (path: number[], action: string, data?: any) => {
    setRootGroup(
      produce(rootGroup, (draft: Group) => {
        if (action === 'addChild') {
          let current = draft;
          for (const i of path) {
            current = current.children[i] as Group;
          }
          current.children.push({
            type: 'keyword',
            word: '',
            rule_op: 'all',
            field: 'title',
          });
        } else if (action === 'convertToGroup') {
          let parent = draft;
          for (let i = 0; i < path.length - 1; i++) {
            parent = parent.children[path[i]] as Group;
          }
          const index = path[path.length - 1];
          const keyword = parent.children[index] as Keyword;
          const newGroup: Group = {
            type: 'group',
            operator: 'AND',
            children: [
              keyword,
              { type: 'keyword', word: '', rule_op: 'all', field: 'title' },
            ],
          };
          parent.children[index] = newGroup;
        } else if (action === 'updateKeyword') {
          let current = draft;
          for (let i = 0; i < path.length - 1; i++) {
            current = current.children[path[i]] as Group;
          }
          current.children[path[path.length - 1]] = {
            ...current.children[path[path.length - 1]],
            ...data,
          };
        } else if (action === 'updateOperator') {
          let current = draft;
          for (const i of path) {
            current = current.children[i] as Group;
          }
          current.operator = data.operator;
        } else if (action === 'deleteNode') {
          if (path.length === 0) return; // Cannot delete root group
          let parent = draft;
          for (let i = 0; i < path.length - 1; i++) {
            parent = parent.children[path[i]] as Group;
          }
          parent.children.splice(path[path.length - 1], 1);

          // Flatten group if it has only one child after delete
          for (let i = 0; i < parent.children.length; i++) {
            const child = parent.children[i];
            if (child.type === 'group' && child.children.length === 1) {
              parent.children[i] = child.children[0];
            }
          }

          // If root group (draft) has only one child and that child is a group, promote it
          if (draft.children.length === 1 && draft.children[0].type === 'group') {
            const promoted = draft.children[0] as Group;
            draft.operator = promoted.operator;
            draft.children = promoted.children;
          }

          // If root group is now empty, reset to initial state
          if (draft.children.length === 0) {
            draft.operator = 'AND';
            draft.children = [{ type: 'keyword', word: '', rule_op: 'all', field: 'title' }];
          }

          // Flatten non-root group if it has only one child (grandparent logic)
          if (path.length >= 2) {
            let grandparent = draft;
            for (let i = 0; i < path.length - 2; i++) {
              grandparent = grandparent.children[path[i]] as Group;
            }
            const parentIndex = path[path.length - 2];
            const maybeGroup = grandparent.children[parentIndex];
            if (
              maybeGroup.type === 'group' &&
              maybeGroup.children.length === 1
            ) {
              grandparent.children[parentIndex] = maybeGroup.children[0];
            }
          }
        }
      })
    );
  };

  // Optional: Function to format state for your endpoint
  // 🔧 Recursively replace `children` ➜ `keywords`
const transformNode = (node: QueryNode): any => {
  if (node.type === "keyword") return node;              // leaf stays unchanged

  const grp = node as Group;
  return {
    type: "group",
    operator: grp.operator,
    keywords: grp.children.map(transformNode),            // recurse
  };
};

// Formats the payload exactly as the Flask endpoint expects
const getQueryForEndpoint = () => ({
  query: {
    group1: transformNode(rootGroup),
  },
});


  // API integration
  const handleSearch = async () => {
    setLoading(true);
    setError(null);
    setHasSearched(false);
    setResults(null);
    try {
      if (!backendPort) throw new Error('Backend port not loaded');
      const response = await fetch(`http://localhost:${backendPort}/api/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(getQueryForEndpoint()),
      });
      if (!response.ok) throw new Error('Failed to fetch search results');
      const data = await response.json();
      let resultsArray = [];
      if (Array.isArray(data)) {
        resultsArray = data;
      } else if (Array.isArray(data.results)) {
        resultsArray = data.results;
      } else if (Array.isArray(data.patents)) {
        resultsArray = data.patents;
      }
      setResults(resultsArray);
    } catch (e: any) {
      setError(e.message || 'Unknown error');
      setResults([]);
    } finally {
      setLoading(false);
      setHasSearched(true);
    }
  };

  return (
    <div style={{ background: '#f5f5f5', borderRadius: 12, padding: 0 }}>
      <GroupComponent group={rootGroup} path={[]} onAction={handleAction} />
      <div style={{ display: 'flex', justifyContent: 'flex-end', width: '100%', marginTop: 10 }}>
        <button
          onClick={handleSearch}
          style={{ ...buttonStyle, minWidth: 120 }}
          disabled={loading}
        >
          Search
        </button>
      </div>
      {error && (
        <div style={{ color: 'red', margin: '10px 0', textAlign: 'right' }}>{error}</div>
      )}
      <div style={{ marginTop: 24 }}>
        <SearchResults hasSearched={hasSearched} results={results || []} loading={loading} />
      </div>
    </div>
  );
};

// Group component
const GroupComponent: React.FC<{
  group: Group;
  path: number[];
  onAction: (path: number[], action: string, data?: any) => void;
}> = ({ group, path, onAction }) => {
  const [operatorDropdownOpen, setOperatorDropdownOpen] = React.useState(false);
  const operatorRef = React.useRef<HTMLDivElement>(null);

  // Close dropdown when clicking outside
  React.useEffect(() => {
    if (!operatorDropdownOpen) return;
    function handleClick(e: MouseEvent) {
      if (
        operatorRef.current &&
        !operatorRef.current.contains(e.target as Node)
      ) {
        setOperatorDropdownOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, [operatorDropdownOpen]);

  return (
    <div style={groupStyle}>
      {/* Visual line and operator label */}
      <div style={groupLineContainerStyle}>
        {group.children.length > 1 && (
          <>
            <div style={groupLineStyle} />
            <div
              style={{ ...groupOperatorLabelStyle, cursor: 'pointer', userSelect: 'none' }}
              title="Click to toggle operator"
              onClick={() =>
                onAction(path, 'updateOperator', { operator: group.operator === 'AND' ? 'OR' : 'AND' })
              }
            >
              {group.operator}
            </div>
            <div style={groupLineStyle} />
          </>
        )}
      </div>
      <div style={{ flex: 1 }}>
        {/* Delete button for subgroups (not for root group) */}
        {path.length > 0 && (
          <button
            onClick={() => onAction(path, 'deleteNode')}
            style={{
              ...deleteButtonStyle,
              position: 'absolute',
              top: 0,
              right: 0,
              transform: 'translate(50%,-50%)',
              fontSize: '1.2em',
              fontWeight: 'bold',
              lineHeight: '1',
              zIndex: 10,
            }}
            title="Delete group"
          >
            ×
          </button>
        )}
        {/* Operator select (keep for logic, but hide visually since operator is now a label) */}

        {group.children.map((child, index) => (
          <div key={index} style={rowStyle}>
            {child.type === 'keyword' ? (
              <KeywordComponent
                keyword={child}
                path={[...path, index]}
                onAction={onAction}
                canDelete={!(path.length === 0 && group.children.length === 1)}
                rootChildrenCount={path.length === 0 ? group.children.length : undefined}
              />
            ) : (
              <GroupComponent
                group={child}
                path={[...path, index]}
                onAction={onAction}
              />
            )}
          </div>
        ))}
        <button onClick={() => onAction(path, 'addChild')} style={buttonStyle}>
          + ADD A ROW
        </button>
      </div>
    </div>
  );
};

// Keyword component
const KeywordComponent: React.FC<{
  keyword: Keyword;
  path: number[];
  onAction: (path: number[], action: string, data?: any) => void;
  canDelete: boolean;
  rootChildrenCount?: number;
}> = ({ keyword, path, onAction, canDelete, rootChildrenCount }) => {
  const [inputFocused, setInputFocused] = React.useState(false);
  const [ruleOpFocused, setRuleOpFocused] = React.useState(false);
  const [fieldFocused, setFieldFocused] = React.useState(false);

  return (
    <div style={rowStyle}>
      <input
        type="text"
        value={keyword.word}
        onChange={(e) =>
          onAction(path, 'updateKeyword', { word: e.target.value })
        }
        style={{
          ...baseInputStyle,
          borderBottom: inputFocused ? '3px solid #BDD248' : '2px solid #bdbdbdbd',
          backgroundColor: inputFocused ? '#E4E4E4' : 'white',
        }}
        placeholder="Enter search term"
        onFocus={() => setInputFocused(true)}
        onBlur={() => setInputFocused(false)}
      />
      <select
        value={keyword.rule_op}
        onChange={(e) =>
          onAction(path, 'updateKeyword', { rule_op: e.target.value })
        }
        style={{
          ...selectStyle,
          backgroundColor: ruleOpFocused ? '#E4E4E4' : 'white',
        }}
        onFocus={() => setRuleOpFocused(true)}
        onBlur={() => setRuleOpFocused(false)}
      >
        <option value="all">all</option>
        <option value="any">any</option>
      </select>
      <select
        value={keyword.field}
        onChange={(e) =>
          onAction(path, 'updateKeyword', { field: e.target.value })
        }
        style={{
          ...selectStyle,
          backgroundColor: fieldFocused ? '#E4E4E4' : 'white',
        }}
        onFocus={() => setFieldFocused(true)}
        onBlur={() => setFieldFocused(false)}
      >
        <option value="title">title</option>
        <option value="abstract">abstract</option>
        <option value="claims">claims</option>
        <option value="title,abstract or claims">title,abstract or claims</option>
        <option value="all text fields">all text fields</option>
        <option value="title or abstract">title or abstract</option>
        <option value="description">description</option>
        <option value="all text fields or names">all text fields or names</option>
        <option value="title , abstract or names">title , abstract or names</option>
      </select>
      {/* Show convert button for all rows if root group has >1 child; hide for initial input if only one row */}
      {((path.length !== 1 || path[0] !== 0) || (rootChildrenCount && rootChildrenCount > 1)) && (
        <button
          onClick={() => onAction(path, 'convertToGroup')}
          style={buttonStyle}
        >
          Convert to AND/OR
        </button>
      )}
      {/* Delete button for each keyword input */}
      {canDelete && (
        <button
          onClick={() => onAction(path, 'deleteNode')}
          style={{
            ...deleteButtonStyle,
            fontSize: '1.2em',
            fontWeight: 'bold',
            lineHeight: '1',
          }}
          title="Delete keyword"
        >
          ×
        </button>
      )}
    </div>
  );
};

export default AdvancedSearchBar;